from Schrodinger_Numerov import calculate_energy_psi 


def main():
	# TODO: Move all parameters except for constants here
	Eall, psiall, xvec, x0vec = calculate_energy_psi(nx0=2)
	print(
		"Energy levels: {}".format(Eall),
		# "Wave functions?: {}".format(psiall),
		# "Positions???: {}".format(xvec),
		# "Initial conditions?: {}".format(x0vec)
	)

if __name__ == '__main__':
	main()